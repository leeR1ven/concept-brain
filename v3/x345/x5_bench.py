"""X5: tick benchmark of the r2b pure-neuron core (numpy on one local core; torch on the A800, batched agents).

Areas at s = 1 (binary units; sizes ASSUMED from r1 sec. 1.6 / 7.5 mapped onto r2b sec. 3, and the X3 hippocampus):
  ear_L1  1,536 thermometer pairs (48 channels x 16 levels x 2), exactly half on (input)
  ear_L2  2,048  <- ear_L1 (16), PFC return (8)               f 0.10
  ear_L3  2,048  <- ear_L2 (16), PFC return (8)               f 0.10
  aud     2,048  <- ear_L3 (64)        plastic                f 0.02   (syllable assemblies)
  voice     256  <- ear_L3 (64)        plastic                f 0.04
  eye_L1  1,024  world eye bits, about 5 % on (input)
  eye_L2  2,048  <- eye_L1 (16), PFC return (8)               f 0.05
  chain copies: 1:1 relays of ear_L2, ear_L3, aud, voice, eye_L2 (8,448 units) to the PFC and to the hippocampus
                (always-on units are not sent); a relay is a copy of the source state, so it costs no synaptic work
  pfc    16,384  <- copies (32, plastic), pfc recurrence (birth 256 + plastic, previous tick), hip (16, plastic)
                    f 0.01, two-layer weights are a per-unit scaling (not timed separately)
  hip    10,000  <- copies (300, birth), birth recurrence (400), plastic recurrence (2,000)    f 0.0064  (X3 base)
  motor   1,024  (512 muscle + 512 inhibitory) <- pfc (128, plastic)                         f 0.05
Per tick: every area computes its drive event-driven from its active inputs (CSR by pre, fixed fan-out), then its
  smoothed quantile threshold (count varies, ceiling 2x); areas whose inputs did not change are skipped;
  Hebbian learning on plastic projections touches only the units that just changed (falling -> rising +,
  rising <-> rising +, rising pre / falling post -), concave growth; every T = 200 ticks all plastic weights
  w <- gamma w and links below w_min are deleted (a full pass, amortised over T).
Stream: speech ticks (a new syllable every 3 ticks: the thermometer channels of the ear move) and silent ticks
  (inputs constant, the PFC and the hippocampus keep running on their recurrence).
Scale s: every unit count x s, fan-ins fixed (r1 rule); option hipC=s also scales the hippocampal recurrence (neuro fix).
Run: python3 -I x5_bench.py numpy [s ...]      python3 x5_bench.py torch [s ...]   (GPU job)
"""
import json
import os
import sys
import time

import numpy as np

T_DEC, GAMMA, W_MIN, ETA, ETA_CO = 200, 0.99, 0.01, 0.3, 0.05


def spec(s, hipC=1.0):
    S = lambda n: int(n * s)
    areas = dict(ear_L1=(S(1536), None), ear_L2=(S(2048), 0.10), ear_L3=(S(2048), 0.10), aud=(S(2048), 0.02),
                 voice=(S(256), 0.04), eye_L1=(S(1024), None), eye_L2=(S(2048), 0.05), pfc=(S(16384), 0.01),
                 hip=(S(10000), 0.0064), motor=(S(1024), 0.05))
    copies = ['ear_L2', 'ear_L3', 'aud', 'voice', 'eye_L2']
    # projections: (src, dst, fan_in, plastic, recurrent=uses previous tick)
    P = [('ear_L1', 'ear_L2', 16, False, False), ('ear_L2', 'ear_L3', 16, False, False),
         ('ear_L3', 'aud', 64, True, False), ('ear_L3', 'voice', 64, True, False),
         ('eye_L1', 'eye_L2', 16, False, False),
         ('copies', 'pfc', 32, True, False), ('pfc', 'pfc', 256, True, True), ('hip', 'pfc', 16, True, True),
         ('pfc', 'ear_L2', 8, True, True), ('pfc', 'ear_L3', 8, True, True), ('pfc', 'eye_L2', 8, True, True),
         ('copies', 'hip', 300, False, False), ('hip', 'hip', 400, False, True),
         ('hip', 'hip', int(2000 * hipC), True, True),
         ('pfc', 'motor', 128, True, True)]
    return areas, copies, P


def build(s, rng, hipC=1.0):
    areas, copies, P = spec(s, hipC)
    n_copy = sum(areas[a][0] for a in copies)
    size = {a: v[0] for a, v in areas.items()}
    size['copies'] = n_copy
    projs = []
    for src, dst, fin, plastic, rec in P:
        ns, nd = size[src], size[dst]
        F = max(1, int(round(fin * nd / ns)))                    # fixed fan-out with the same edge count
        idx_t = np.int32
        post = rng.integers(0, nd, size=ns * F).astype(idx_t)
        w = rng.uniform(0.05, 0.20, ns * F).astype(np.float32) * (1.0 / max(1.0, fin / 16))
        projs.append(dict(src=src, dst=dst, F=F, post=post, w=w, plastic=plastic, rec=rec, n_src=ns, n_dst=nd))
    n_syn = sum(len(p['post']) for p in projs)
    n_units = sum(size[a] for a in areas) + 2 * n_copy
    return areas, copies, projs, size, n_syn, n_units


# ----------------------------------------------------------------------------------------------- numpy, 1 agent
class NumpyCore:
    def __init__(self, s, seed=0, hipC=1.0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.areas, self.copies, self.projs, self.size, self.n_syn, self.n_units = build(s, rng, hipC)
        self.state = {a: np.zeros(n, bool) for a, n in self.size.items()}
        self.prev = {a: v.copy() for a, v in self.state.items()}
        self.theta = {a: None for a in self.areas}
        self.t = 0
        self.ch = 48
        self.levels = rng.integers(0, 16, int(self.ch * s))
        self.eye = rng.random(self.size['eye_L1']) < 0.05
        self.n_ear = self.size['ear_L1']

    def thermo(self):
        x = np.zeros(self.n_ear, bool)
        L = self.levels
        j = np.arange(len(L))[:, None] * 32 + np.arange(16)[None, :] * 2
        on = np.where(np.arange(16)[None, :] < L[:, None], j, j + 1)
        x[on.ravel()] = True
        return x

    def src_state(self, name, prev):
        st = self.prev if prev else self.state
        if name == 'copies':
            return np.concatenate([st[a] for a in self.copies])
        return st[name]

    def drive(self, p, x):
        act = np.flatnonzero(x)
        if len(act) == 0:
            return None
        e = (act[:, None] * p['F'] + np.arange(p['F'])[None, :]).ravel()
        return np.bincount(p['post'][e], weights=p['w'][e], minlength=p['n_dst'])

    def fire(self, a, d):
        n = len(d); f = self.areas[a][1]; k = max(1, int(round(f * n)))
        part = np.partition(d, (n - 2 * k, n - k))
        q, cap = part[n - k], part[n - 2 * k]
        th = self.theta[a]
        th = q if th is None else th + 0.3 * (q - th)
        self.theta[a] = th
        return (d >= max(th, cap, 1e-9)) & (d > 0)

    def learn(self, p, sx_prev, sx, dx_prev, dx):
        rise_s, fall_s = np.flatnonzero(sx & ~sx_prev), np.flatnonzero(sx_prev & ~sx)
        rise_d, fall_d = dx & ~dx_prev, dx_prev & ~dx
        if not rise_d.any() and not fall_d.any():
            return
        for pre, mask, eta, sign in ((fall_s, rise_d, ETA, 1.0), (rise_s, rise_d, ETA_CO, 1.0),
                                     (rise_s, fall_d, ETA, -1.0)):
            if len(pre) == 0:
                continue
            e = (pre[:, None] * p['F'] + np.arange(p['F'])[None, :]).ravel()
            m = mask[p['post'][e]]
            if not m.any():
                continue
            e = e[m]
            w = p['w'][e]
            p['w'][e] = np.maximum(0.0, w + sign * eta / (1.0 + w))

    def tick(self, speech):
        self.prev = {a: v.copy() for a, v in self.state.items()}
        if speech and self.t % 3 == 0:
            ch = self.rng.integers(0, len(self.levels), len(self.levels) // 2)
            self.levels[ch] = self.rng.integers(0, 16, len(ch))
        self.state['ear_L1'] = self.thermo()
        self.state['eye_L1'] = self.eye
        order = ['ear_L2', 'ear_L3', 'aud', 'voice', 'eye_L2', 'pfc', 'hip', 'motor']
        for a in order:
            d = np.zeros(self.size[a])
            for p in self.projs:
                if p['dst'] == a:
                    dd = self.drive(p, self.src_state(p['src'], p['rec']))
                    if dd is not None:
                        d += dd
            self.state[a] = self.fire(a, d)
        for p in self.projs:
            if p['plastic']:
                self.learn(p, self.src_state(p['src'], True), self.src_state(p['src'], False),
                           self.prev[p['dst']], self.state[p['dst']])
        self.t += 1
        if self.t % T_DEC == 0:
            self.decay()

    def decay(self):
        for p in self.projs:
            if p['plastic']:
                p['w'] *= GAMMA
                p['w'][p['w'] < W_MIN] = 0.0


def bench_area_numpy(s, area, n_ticks=200, hipC=1.0, fan_cap=1000):
    """one area alone: its incoming projections, random source states at the areas' activity, a fraction of the
    active source units changing per tick (speech 0.3, silent 0.1 for the PFC / hippocampus, else 0); edge-
    proportional work is measured at fan-out <= fan_cap and scaled up linearly (memory cap 600 MB)"""
    rng = np.random.default_rng(1)
    areas, copies, P = spec(s, hipC)
    size = {a: v[0] for a, v in areas.items()}; size['copies'] = sum(size[a] for a in copies)
    frac = {a: (v[1] if v[1] else (0.5 if a == 'ear_L1' else 0.05)) for a, v in areas.items()}
    frac['copies'] = sum(size[a] * frac[a] for a in copies) / size['copies']
    ins = [p for p in P if p[1] == area]
    projs, scale = [], []
    for src, dst, fin, plastic, rec in ins:
        ns, nd = size[src], size[dst]
        F = max(1, int(round(fin * nd / ns)))
        Fm = min(F, fan_cap)
        projs.append(dict(src=src, F=Fm, post=rng.integers(0, nd, ns * Fm).astype(np.int32),
                          w=rng.uniform(0.05, 0.2, ns * Fm).astype(np.float32), plastic=plastic, n_dst=nd))
        scale.append(F / Fm)
    core = NumpyCore.__new__(NumpyCore)
    core.areas = areas; core.theta = {area: None}
    st = {p['src']: rng.random(size[p['src']]) < frac[p['src']] for p in projs}
    dst = np.zeros(size[area], bool)
    res = {}
    for kind, ch in (('speech', 0.3), ('silent', 0.1 if area in ('pfc', 'hip', 'motor') else 0.0)):
        t_drv = t_thr = t_lrn = 0.0
        for _ in range(n_ticks):
            prev_st = {k: v.copy() for k, v in st.items()}
            for k, v in st.items():
                if ch and (k not in ('ear_L1', 'eye_L1') or kind == 'speech'):
                    on = np.flatnonzero(v); off = np.flatnonzero(~v)
                    m = max(1, int(ch * len(on)))
                    v[rng.choice(on, m, replace=False)] = False; v[rng.choice(off, m, replace=False)] = True
            t0 = time.perf_counter()
            d = np.zeros(size[area])
            for p, sc in zip(projs, scale):
                dd = core.drive(p, st[p['src']])
                if dd is not None:
                    d += dd
            t1 = time.perf_counter()
            prev_dst = dst; dst = core.fire(area, d)
            t2 = time.perf_counter()
            for p in projs:
                if p['plastic']:
                    core.learn(p, prev_st[p['src']], st[p['src']], prev_dst, dst)
            t3 = time.perf_counter()
            esc = float(np.mean(scale)) if scale else 1.0
            t_drv += (t1 - t0) * esc; t_thr += t2 - t1; t_lrn += (t3 - t2) * esc
        res[kind] = dict(drive_ms=round(1000 * t_drv / n_ticks, 3), thr_ms=round(1000 * t_thr / n_ticks, 3),
                         learn_ms=round(1000 * t_lrn / n_ticks, 3))
    ts = time.perf_counter()
    for p in projs:
        if p['plastic']:
            p['w'] *= GAMMA; p['w'][p['w'] < W_MIN] = 0.0
    res['decay_pass_ms'] = round(1000 * (time.perf_counter() - ts) * (float(np.mean(scale)) if scale else 1), 2)
    res['synapses'] = int(sum(len(p['post']) * sc for p, sc in zip(projs, scale)))
    return res


def bench_numpy(s, n_ticks=300, hipC=1.0):
    t0 = time.time()
    core = NumpyCore(s, hipC=hipC)
    build_s = time.time() - t0
    for _ in range(20):
        core.tick(True)
    res = {}
    for kind, speech in (('speech', True), ('silent', False)):
        ts = time.time()
        for _ in range(n_ticks):
            if core.t % T_DEC == T_DEC - 1:
                core.t += 1                                    # time the decay pass separately
            core.tick(speech)
        res[kind + '_ms'] = round(1000 * (time.time() - ts) / n_ticks, 2)
    ts = time.time(); core.decay(); res['decay_pass_ms'] = round(1000 * (time.time() - ts), 1)
    res['decay_amortised_ms'] = round(res['decay_pass_ms'] / T_DEC, 3)
    act = {a: int(core.state[a].sum()) for a in core.areas}
    import resource
    return dict(backend='numpy', s=s, hipC=hipC, units=core.n_units, synapses=core.n_syn, build_s=round(build_s, 1),
                active=act, rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), **res)


# ----------------------------------------------------------------------------------------------- torch, B agents
def bench_torch(s, B, n_ticks=200, hipC=1.0, dev='cuda'):
    import torch
    torch.set_num_threads(1)
    g = torch.Generator(device='cpu').manual_seed(0)
    rng = np.random.default_rng(0)
    areas, copies, projs, size, n_syn, n_units = build(s, rng, hipC)
    order = ['ear_L2', 'ear_L3', 'aud', 'voice', 'eye_L2', 'pfc', 'hip', 'motor']
    T = {}
    for p in projs:                                            # shared structure, per-agent weights
        p['post_t'] = torch.from_numpy(p['post'].astype(np.int64)).to(dev)
        p['w_t'] = torch.from_numpy(p['w']).to(dev).unsqueeze(0).repeat(B, 1).contiguous()
        p.pop('post'); p.pop('w')
    if dev == 'cuda':
        torch.cuda.synchronize()
    mem_w = sum(p['w_t'].numel() * 4 for p in projs) / 2 ** 30
    state = {a: torch.zeros(B, n, dtype=torch.bool, device=dev) for a, n in size.items()}
    theta = {a: None for a in areas}
    n_ch = int(48 * s)
    levels = torch.randint(0, 16, (B, n_ch), device=dev)
    eye = torch.rand(B, size['eye_L1'], device=dev) < 0.05
    ar16 = torch.arange(16, device=dev)

    def thermo():
        on = (ar16[None, None, :] < levels[:, :, None])                   # B x ch x 16
        x = torch.stack([on, ~on], -1).reshape(B, -1)
        return x[:, :size['ear_L1']]

    def src(name, st):
        return torch.cat([st[a] for a in copies], 1) if name == 'copies' else st[name]

    def drive(p, x, out):
        b, i = torch.nonzero(x, as_tuple=True)
        if b.numel() == 0:
            return
        F = p['F']
        e = (i[:, None] * F + torch.arange(F, device=dev)[None, :]).reshape(-1)
        bb = b.repeat_interleave(F)
        w = p['w_t'][bb, e]
        out.view(-1).index_add_(0, bb * p['n_dst'] + p['post_t'][e], w)

    def fire(a, d):
        n = d.shape[1]; f = areas[a][1]; k = max(1, int(round(f * n)))
        v = torch.topk(d, min(n, 2 * k), dim=1).values
        q, cap = v[:, k - 1], v[:, -1]
        th = theta[a]
        th = q if th is None else th + 0.3 * (q - th)
        theta[a] = th
        return (d >= torch.maximum(torch.maximum(th, cap), torch.full_like(th, 1e-9))[:, None]) & (d > 0)

    def learn(p, sxp, sx, dxp, dx):
        rise_d, fall_d = dx & ~dxp, dxp & ~dx
        F = p['F']
        for pre_m, mask, eta, sign in ((sxp & ~sx, rise_d, ETA, 1.0), (sx & ~sxp, rise_d, ETA_CO, 1.0),
                                       (sx & ~sxp, fall_d, ETA, -1.0)):
            b, i = torch.nonzero(pre_m, as_tuple=True)
            if b.numel() == 0:
                continue
            e = (i[:, None] * F + torch.arange(F, device=dev)[None, :]).reshape(-1)
            bb = b.repeat_interleave(F)
            m = mask[bb, p['post_t'][e]]
            e, bb = e[m], bb[m]
            if e.numel() == 0:
                continue
            w = p['w_t'][bb, e]
            p['w_t'][bb, e] = torch.clamp(w + sign * eta / (1.0 + w), min=0.0)

    def tick(t, speech):
        nonlocal levels
        prev = {a: v for a, v in state.items()}
        if speech and t % 3 == 0:
            msk = torch.rand(B, n_ch, device=dev) < 0.5
            levels = torch.where(msk, torch.randint(0, 16, (B, n_ch), device=dev), levels)
        state['ear_L1'] = thermo(); state['eye_L1'] = eye
        for a in order:
            ins = [p for p in projs if p['dst'] == a]
            d = torch.zeros(B, size[a], device=dev)
            for p in ins:
                drive(p, src(p['src'], prev if p['rec'] else state), d)
            state[a] = fire(a, d)
        for p in projs:
            if p['plastic']:
                learn(p, src(p['src'], prev), src(p['src'], state), prev[p['dst']], state[p['dst']])

    def decay():
        for p in projs:
            if p['plastic']:
                p['w_t'].mul_(GAMMA)
                p['w_t'][p['w_t'] < W_MIN] = 0.0

    for t in range(20):
        tick(t, True)
    if dev == 'cuda':
        torch.cuda.synchronize()
    res = {}
    for kind, speech in (('speech', True), ('silent', False)):
        ts = time.time()
        for t in range(n_ticks):
            tick(t, speech)
        if dev == 'cuda':
            torch.cuda.synchronize()
        res[kind + '_ms'] = round(1000 * (time.time() - ts) / n_ticks, 3)
    ts = time.time(); decay()
    if dev == 'cuda':
        torch.cuda.synchronize()
    res['decay_pass_ms'] = round(1000 * (time.time() - ts), 2)
    res['decay_amortised_ms'] = round(res['decay_pass_ms'] / T_DEC, 4)
    res['speech_ms_per_agent'] = round(res['speech_ms'] / B, 4)
    res['silent_ms_per_agent'] = round(res['silent_ms'] / B, 4)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == 'cuda' else 0
    return dict(backend='torch-' + dev, s=s, B=B, hipC=hipC, units=n_units, synapses=n_syn,
                weights_gb=round(mem_w, 2), peak_gpu_gb=round(peak, 2), **res)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'numpy'
    out = []
    if mode == 'areas':
        for a in sys.argv[2:] or ['1']:
            sv, _, hc = a.partition(':')
            sv, hc = float(sv), float(hc or 1.0)
            tot = dict(s=sv, hipC=hc, areas={})
            for ar in ['ear_L2', 'ear_L3', 'aud', 'voice', 'eye_L2', 'pfc', 'hip', 'motor']:
                tot["areas"][ar] = bench_area_numpy(sv, ar, hipC=hc, fan_cap=1000 if sv < 2 else 500)
            for kind in ('speech', 'silent'):
                tot[kind + '_ms'] = round(sum(sum(v[kind].values()) for v in tot['areas'].values()), 2)
            tot['decay_amortised_ms'] = round(sum(v['decay_pass_ms'] for v in tot['areas'].values()) / T_DEC, 3)
            tot['synapses'] = sum(v['synapses'] for v in tot['areas'].values())
            import resource
            tot['rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            print(json.dumps(tot), flush=True); out.append(tot)
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'x5_areas.json'), 'w'), indent=1)
    elif mode == 'numpy':
        for a in sys.argv[2:] or ['1']:
            s, _, hc = a.partition(':')
            r = bench_numpy(float(s), hipC=float(hc or 1.0))
            print(json.dumps(r), flush=True); out.append(r)
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'x5_numpy.json'), 'w'), indent=1)
    else:
        import gc
        import traceback
        import torch
        print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)
        try:                                                       # CPU smoke first: catches bugs in seconds
            print(json.dumps(bench_torch(0.25, 2, n_ticks=10, dev='cpu')), flush=True)
        except Exception:
            traceback.print_exc()
        torch.cuda.set_per_process_memory_fraction(0.22)
        for s, B, hc in [(1, 1, 1), (1, 16, 1), (1, 64, 1), (4, 1, 1), (4, 16, 1), (4, 64, 1), (1, 64, 4)]:
            try:
                r = bench_torch(s, B, hipC=hc)
            except Exception as e:                                 # out of memory under the 18 GB cap, or a bug
                r = dict(backend='torch-cuda', s=s, B=B, hipC=hc, error=type(e).__name__ + ': ' + str(e)[:300])
                traceback.print_exc()
            print(json.dumps(r), flush=True); out.append(r)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        json.dump(out, open('x5_torch.json', 'w'), indent=1)
