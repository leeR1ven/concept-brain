"""The X10 brain: sound area -> 1:1 chain pairs -> PFC and hippocampus; returns 1:1 back; motor area.
Brain.step(a0, r) is the only entry point: it receives the A0 bit vector and r, the innate value signal (0.0
unless the reward tool is on, stage b2 [R6]).

Build order (birth): record a birth stream with the child silent (no speech wiring yet) -> calibrate the sound thresholds ->
chain band -> build PFC, hippocampus -> calibrate their fixed thresholds on the time-parallel birth chain stream."""
import collections

import numpy as np
import torch

from common import DEV, gammas, log
from hippocampus import Hippocampus
from motor import Motor
from pfc import PFC
from sound import SoundArea, touch_params
from world_adapter import Skin


class Chains:
    """[S5, C1-C3] chain sources: chain-bearing units of A1..A4 (birth rate in the band), every force neuron, every
    stop neuron. Copy j of the PFC and copy j of the hippocampus both belong to source j (forward), and both return
    to source j only (1:1). The input layer A0 and the blocking neurons have no chain pair."""

    def __init__(self, sound, motor, rates, bands, seed, touch=None, trates=None, tbands=None):
        """bands: per sound layer (low, high), already resolved (shallow band or depth-relative cut; None = every
        unit). Force and stop neurons are all chained; blocking neurons never (2.7)."""
        self.sound_idx = [torch.arange(r.numel(), device=DEV) if b is None else
                          torch.nonzero((r >= b[0]) & (r <= b[1])).squeeze(1) for r, b in zip(rates, bands)]
        segs, pos = [], []
        a = 0
        for l, ix in enumerate(self.sound_idx):
            segs.append(('sound', l, a, a + len(ix)))
            pos.append(sound.pos[l][ix])
            a += len(ix)
        g = torch.Generator(device=DEV)
        g.manual_seed(seed + 300)
        segs.append(('force', 0, a, a + motor.nf))
        pos.append(torch.rand((motor.nf, 2), generator=g, device=DEV))
        a += motor.nf
        segs.append(('stop', 0, a, a + motor.ns))
        pos.append(torch.rand((motor.ns, 2), generator=g, device=DEV))
        a += motor.ns
        self.touch_idx = []                                     # [T3] touch layers' chain units (after the motor)
        if touch is not None:
            self.touch_idx = [torch.arange(r.numel(), device=DEV) if b is None else
                              torch.nonzero((r >= b[0]) & (r <= b[1])).squeeze(1) for r, b in zip(trates, tbands)]
            for l, ix in enumerate(self.touch_idx):
                segs.append(('touch', l, a, a + len(ix)))
                pos.append(touch.pos[l][ix])
                a += len(ix)
        self.segs, self.n = segs, a
        jit = 0.005 * torch.randn((a, 2), generator=g, device=DEV)
        self.pos = (torch.cat(pos) + jit).clamp(0.0, 1.0)
        # explicit source table (area code, layer, unit) for the 1:1 checks
        area = torch.zeros(a, dtype=torch.int64, device=DEV)
        layer = torch.zeros(a, dtype=torch.int64, device=DEV)
        unit = torch.zeros(a, dtype=torch.int64, device=DEV)
        for kind, l, s0, s1 in segs:
            area[s0:s1] = {'sound': 0, 'force': 1, 'stop': 2, 'touch': 3}[kind]
            layer[s0:s1] = l
            unit[s0:s1] = (self.sound_idx[l] if kind == 'sound' else self.touch_idx[l] if kind == 'touch'
                           else torch.arange(s1 - s0, device=DEV))
        self.fwd_src = torch.stack([area, layer, unit], 1)      # forward chain j comes from this source
        self.ret_dst = self.fwd_src.clone()                     # return chain j goes back to this unit

    def gather(self, sound_state, force, stop, touch_state=None):
        parts = [sound_state[l][ix] for l, ix in enumerate(self.sound_idx)]
        tparts = [touch_state[l][ix] for l, ix in enumerate(self.touch_idx)] if self.touch_idx else []
        return torch.cat(parts + [force, stop] + tparts)

    def scatter(self, R, sound_n, nf, ns, touch_n=None):
        """returns R (bool [n]) to its own sources: per sound layer float [n_l], force float [nf], stop bool [ns],
        per touch layer float [n_l]"""
        out, tout = [], []
        for kind, l, s0, s1 in self.segs:
            if kind == 'sound':
                v = torch.zeros(sound_n[l], device=DEV)
                v[self.sound_idx[l]] = R[s0:s1].float()
                out.append(v)
            elif kind == 'force':
                rf = R[s0:s1].float()
            elif kind == 'stop':
                rs = R[s0:s1]
            else:
                v = torch.zeros(touch_n[l], device=DEV)
                v[self.touch_idx[l]] = R[s0:s1].float()
                tout.append(v)
        return out, rf, rs, tout


class Brain:
    def __init__(self, c, seed, coch, birth_A0, birth_speech=None):
        """birth_A0: bool numpy [T, n0], the birth period heard by the silent newborn. birth_speech (optional, report
        only, never given to the brain's dynamics): which birth ticks had adult speech."""
        self.c, self.seed = c, seed
        self.gam = gammas(c)
        self.sound = SoundArea(coch, c)
        A0 = torch.as_tensor(birth_A0, device=DEV)
        states = self.sound.calibrate(A0)
        rates = [st.float().mean(0) for st in states]
        self.birth_rates = rates
        self.motor = Motor(c, coch, self.gam)
        bs = c['band_sound'] if c['band_sound'] is not None else c['band']
        shallow = list(bs) if isinstance(bs[0], (list, tuple)) else [tuple(bs)] * len(rates)
        bands = []
        for l in range(len(rates)):
            if l < c['band_depth_sound']:
                bands.append(tuple(shallow[l]))
            elif c['band_rel'] > 0:
                bands.append((c['band_rel'] * c['sound_rates'][l], c['band'][1]))
            else:
                bands.append(None)
        self.touch, self.skin, t_states, t_rates, t_bands = None, None, None, None, None
        if c['touch']:                                      # [T1-T5] touch area incl. muscle force (designer 11-02)
            self.skin = Skin(c, self.motor.m)
            self.touch = SoundArea(self.skin, c, touch_params(c))
            TB = self.skin.birth_stream(A0.shape[0], coch.ph.need_mat, seed)
            t_states = self.touch.calibrate(TB)
            t_rates = [st.float().mean(0) for st in t_states]
            t_bands = [tuple(c['band_touch'])] * len(t_rates)
            self.touch_rates_birth = t_rates
        self.motor_rates_birth = self.motor_birth_rates(A0)
        lo_m, hi_m = c['band_motor']
        for k in ('force', 'stop'):                         # the motor band, applied to force and stop neurons
            r = self.motor_rates_birth[k]
            assert bool(((r >= lo_m) & (r <= hi_m)).all()), f'motor band cuts {k} neurons: not supported yet'
        self.chains = Chains(self.sound, self.motor, rates, bands, seed, self.touch, t_rates, t_bands)
        self.chain_report = self.make_chain_report(states, rates, bands, birth_speech, t_states, t_rates, t_bands)
        del t_states
        self.n1 = self.chains.n
        log(f'chains: {[len(ix) for ix in self.chains.sound_idx]} sound + {self.motor.nf} force + {self.motor.ns} stop '
            f'+ {[len(ix) for ix in self.chains.touch_idx]} touch = {self.n1}')
        self.pfc = PFC(c, self.chains.pos, self.gam, seed)
        self.hip = Hippocampus(c, self.chains.pos, self.gam, seed) if c['hip'] else None
        # time-parallel birth chain stream: sound states at t-1 (all layers taken from the same tick), stop neurons
        # from the brake reflex, force neurons silent (no G1 yet)
        T = A0.shape[0]
        brake = A0[:, self.motor.brake_lines].float().sum(1, keepdim=True) / self.motor.n_brake
        stop = brake >= self.motor.thr_s[None]
        force = torch.zeros((T, self.motor.nf), dtype=torch.bool, device=DEV)
        tparts = []
        if self.touch is not None:                          # the newborn does not move: touch at rest
            rest = self.rest_touch_state()
            tparts = [rest[l][ix][None].expand(T, -1) for l, ix in enumerate(self.chains.touch_idx)]
        X = torch.cat([st[:, ix] for st, ix in zip(states, self.chains.sound_idx)] + [force, stop] + tparts, 1)
        X = torch.cat([torch.zeros_like(X[:1]), X[:-1]])                    # chain relay: one tick
        X = X[-c['calib_ticks']:]                                            # the latest birth ticks only
        del states
        self.calib = dict(pfc_counts=self.pfc.calibrate(X), pfc_theta=self.pfc.theta, pfc_g=self.pfc.g_l)
        if self.hip is not None:
            self.calib['hip'] = self.hip.calibrate(X)
        self.birth_X_counts = X.float().sum(1).cpu().numpy()
        del X
        n = self.sound.n
        self.ret_sound = [torch.zeros(k, device=DEV) for k in n]
        self.ret_force = torch.zeros(self.motor.nf, device=DEV)
        self.ret_stop = torch.zeros(self.motor.ns, dtype=torch.bool, device=DEV)
        self.ret_touch = [torch.zeros(k, device=DEV) for k in self.touch.n] if self.touch is not None else []
        if self.touch is not None:
            self.touch.state = self.rest_touch_state()
        self.learning = False
        self.prof = None                                     # a common.Prof when profiling
        self.scaffold = None                                 # teacher program of the learned-speech arm (stage b)
        self.t = 0
        self.force_int = None
        self.n0 = coch.n
        self.rlog = collections.deque()                      # [R6] recent applied changes (reward tool)
        if c['reward']:
            for tb in self.plastic():
                tb.track = True

    # ------------------------------------------------------------------ online birth calibration
    def calibrate_online(self, A0_stream, every=50, rate=0.2):
        """Replay the birth stream (newborn silent, learning off) with this configuration's birth loop and birth
        lateral links on, and move each PFC layer's fixed threshold (and Hs's) until the mean counts follow the
        layer below (Hs: kh). The thresholds are then frozen: a birth calibration by the experimenter, not a
        mechanism of the running brain. Returns the trajectory of mean counts."""
        was = self.learning
        self.learning = False
        sig = [max(abs(t), 1e-3) for t in self.pfc.theta]
        sig_h = max(abs(self.hip.theta_hs), 1e-3) if self.hip is not None else 0.0
        acc = torch.zeros(7, device=DEV)
        traj = []
        k_ctx = self.c['ctx_k']
        for t in range(len(A0_stream)):
            self.step(torch.as_tensor(A0_stream[t], device=DEV), 0.0)
            p = self.pfc
            v = torch.stack([p.L1.on.sum(), p.L2.on.sum(), p.L3.on.sum(), p.L4.on.sum(),
                             self.hip.Hs.on.sum() if self.hip is not None else torch.zeros((), device=DEV),
                             p.L1ext.on.sum(), p.x_int.sum()])
            acc += v.float()
            if (t + 1) % every == 0:
                m = (acc / every).tolist()
                for l in (1, 2, 3):
                    self.pfc.theta[l] += rate * sig[l] * (m[l] - m[l - 1]) / max(m[l - 1], 1.0)
                if self.hip is not None:
                    self.hip.theta_hs += rate * sig_h * (m[4] - self.c['kh']) / self.c['kh']
                if k_ctx > 0:                                   # excitability test: L1int ~ k x L1ext
                    tgt = k_ctx * max(m[5], 1.0)
                    self.pfc.theta[0] -= rate * 0.5 * (tgt - m[6]) / tgt
                    if self.pfc.theta[0] < 0.05:                # theta_1 at its floor: raise the loop gain
                        self.pfc.theta[0] = 0.05
                        if m[6] < tgt:
                            self.pfc.loop.w = self.pfc.loop.w * 1.25
                traj.append([round(x, 1) for x in m])
                acc.zero_()
        self.learning = was
        self.calib['online'] = dict(theta=list(self.pfc.theta), theta_hs=self.hip.theta_hs if self.hip else None,
                                    last_counts=traj[-1] if traj else None, first_counts=traj[0] if traj else None)
        return traj

    def rest_touch_state(self, ticks=6):
        """the touch layers' states when no muscle moves (no returns): the newborn's touch input"""
        rest = self.skin.code(torch.zeros(24, device=DEV))
        st = [torch.zeros(k, dtype=torch.bool, device=DEV) for k in self.touch.n]
        self.touch.state = st
        zero = [torch.zeros(k, device=DEV) for k in self.touch.n]
        for _ in range(ticks):
            st = self.touch.step(rest, zero)
        return [x.clone() for x in st]

    # ------------------------------------------------------------------ chain bands and their report
    def motor_birth_rates(self, A0):
        mot = self.motor
        brake = A0[:, mot.brake_lines].float().sum(1, keepdim=True) / mot.n_brake
        stop = (brake >= mot.thr_s[None]).float().mean(0)
        block = torch.zeros(mot.nb, device=DEV)
        block[:mot.n_tonic] = 1.0                                           # tonic; ladder units need muscle drive
        return dict(force=torch.zeros(mot.nf, device=DEV), stop=stop, block=block)

    def make_chain_report(self, states, rates, bands, speech, t_states=None, t_rates=None, t_bands=None):
        """per area and layer: share of units in the band, copies sent to PFC L1 and to the hippocampus (1:1, the
        same number), lit chain units in silence and in adult speech over the birth stream"""
        rep = []
        lab = None if speech is None else torch.as_tensor(np.asarray(speech), device=DEV)
        for l, (st, r, ix) in enumerate(zip(states, rates, self.chains.sound_idx)):
            lo, hi = bands[l] if bands[l] is not None else self.c['band']
            rc = r[ix]
            row = dict(area='sound', layer=f'A{l + 1}', units=int(r.numel()),
                       cut=list(bands[l]) if bands[l] is not None else 'none (every unit chained)',
                       in_band=float(((r >= lo) & (r <= hi)).float().mean()),
                       never_on=float((r < lo).float().mean()), always_on=float((r > hi).float().mean()),
                       chained=int(len(ix)), chained_in_band=int(((rc >= lo) & (rc <= hi)).sum()),
                       chained_never_on=int((rc < lo).sum()), chained_always_on=int((rc > hi).sum()),
                       copies_pfc=int(len(ix)), copies_hip=int(len(ix)) if self.c['hip'] else 0)
            if lab is not None and len(lab) == st.shape[0]:
                cnt = st[:, ix].float().sum(1)
                row.update(lit_silence=float(cnt[~lab].mean()), lit_speech=float(cnt[lab].mean()))
            rep.append(row)
        if t_rates is not None:
            for l, (r, ix) in enumerate(zip(t_rates, self.chains.touch_idx)):
                lo, hi = t_bands[l]
                rep.append(dict(area='touch', layer=f'T{l + 1}', units=int(r.numel()), cut=list(t_bands[l]),
                                in_band=float(((r >= lo) & (r <= hi)).float().mean()),
                                never_on=float((r < lo).float().mean()), always_on=float((r > hi).float().mean()),
                                chained=int(len(ix)), copies_pfc=int(len(ix)), copies_hip=int(len(ix)) if self.c['hip'] else 0,
                                birth='spontaneous twitches (touch only)'))
        mr = self.motor_rates_birth
        lo, hi = self.c['band_motor']
        for k, n in (('force', self.motor.nf), ('stop', self.motor.ns), ('block', self.motor.nb)):
            r = mr[k]
            inb = (r >= lo) & (r <= hi)
            sent = n if k != 'block' else 0
            rep.append(dict(area='motor', layer=k, units=n, band=[lo, hi], in_band=float(inb.float().mean()),
                            copies_pfc=sent, copies_hip=sent if self.c['hip'] else 0, birth_rate_mean=float(r.mean())))
        return rep

    # ------------------------------------------------------------------ registry helpers
    def tables(self):
        out = list(self.sound.layers) + (list(self.touch.layers) if self.touch is not None else []) + \
            self.pfc.tables() + self.motor.tables()
        if self.hip is not None:
            out += self.hip.tables()
        return out

    def plastic(self):
        out = self.pfc.plastic() + self.motor.tables()     # motor: MC pattern pools only
        if self.hip is not None:
            out += self.hip.plastic()
        return out

    def pops(self):
        out = self.pfc.pops() + self.motor.pops()
        if self.hip is not None:
            out += self.hip.pops()
        return out

    # ------------------------------------------------------------------ one tick
    def step(self, a0, r=0.0):
        assert isinstance(a0, torch.Tensor) and a0.dtype == torch.bool and tuple(a0.shape) == (self.n0,), \
            'the brain receives only the A0 bit vector'
        assert isinstance(r, float), 'r is the innate value signal (float)'
        assert r == 0.0 or self.c['reward'], 'reward is off in this configuration [R5]'
        ch = self.chains
        P = self.prof
        P and P.mark()
        x_ext = ch.gather(self.sound.state, self.motor.force.on, self.motor.stop.on,
                          self.touch.state if self.touch is not None else None)          # sources at t-1
        s_new = self.sound.step(a0, self.ret_sound)                                        # returns from t-1
        P and P.lap('sound+chains')
        muscles = self.motor.step(a0, self.ret_force, self.ret_stop, self.scaffold)
        if self.touch is not None:                          # [T4] the body's actual force, felt in the same tick
            cnt = torch.zeros(24, device=DEV).index_add_(0, self.motor.muscle_of, self.motor.force.on.float())
            self.touch.step(self.skin.code(self.skin.force(cnt)), self.ret_touch)
        P and P.lap('motor')
        x_int = self.pfc.step(x_ext, self.force_int)
        P and P.lap('pfc')
        R = x_int
        if self.hip is not None:
            R = R | self.hip.step(x_ext)
        P and P.lap('hip')
        self.ret_sound, self.ret_force, self.ret_stop, self.ret_touch = ch.scatter(
            R, self.sound.n, self.motor.nf, self.motor.ns, self.touch.n if self.touch is not None else None)
        P and P.lap('returns')
        if self.learning:
            self.pfc.learn()
            P and P.lap('learn pfc')
            self.motor.learn()
            if self.hip is not None:
                self.hip.learn()
            P and P.lap('learn motor+hip')
            if self.c['reward']:                            # [R6] keep this tick's applied changes
                entries = []
                for tb in self.plastic():
                    entries += [(tb, rows, d) for rows, d in tb.tick_log]
                    tb.tick_log = []
                self.rlog.append((self.t, entries))
                while self.rlog and self.t - self.rlog[0][0] >= self.c['r_win']:
                    self.rlog.popleft()
            if (self.t + 1) % self.c['decay_every'] == 0:
                for tb in self.plastic():
                    tb.decay()
                P and P.lap('decay')
        if r != 0.0:
            self.reward(r)
        self.t += 1
        return muscles

    # ------------------------------------------------------------------ the reward tool [R6, ARCH 2.8]
    def reward(self, r):
        """ARCH 2.8 "奖励只是工具：在'这一刻算数'时，放大刚才一起亮的连接的变化。高兴时放大这些变化；难过时把这些变化
        反过来，并让当时的说话状态连到运动区的停止神经元上。" r > 0 (pleased): every applied change of the last r_win
        ticks is added again x r_pleased x r, fading linearly with age; r < 0 (upset): added again x -(1 + r_upset) x |r|
        (net reversal), and the speech state of this moment (the L4 units on now) strengthens its existing learned
        L4 -> L1 links onto the stop neurons' copies by the rule's own step x r_stop x |r| (an ordinary acquired link:
        stop copies lit internally return 1:1 to the stop neurons). Form and parameters from X8 (win 24, pleased x2,
        upset net x-1)."""
        c = self.c
        gain = c['r_pleased'] * r if r > 0 else -(1.0 + c['r_upset']) * (-r)
        for t0, entries in self.rlog:
            f = gain * (1.0 - (self.t - t0) / c['r_win'])
            for tb, rows, d in entries:
                tb._apply(rows, f * d, log=False)
        self.rlog.clear()
        if r < 0:
            L = self.pfc.L41
            seg = [sg for sg in self.chains.segs if sg[0] == 'stop'][0]
            rows = torch.nonzero(self.pfc.L4.on).squeeze(1)
            if rows.numel():
                ix = L.idx[rows].long()
                to_stop = ((ix >= seg[2]) & (ix < seg[3])).float()
                dw = c['eta'] * c['r_stop'] * (-r) * L.plast(L.w[rows]) * to_stop
                L._apply(rows, dw, log=False)

    # ------------------------------------------------------------------ probes: neuron states only
    def snapshot(self):
        return dict(sound=[x.clone() for x in self.sound.state], ret=[x.clone() for x in self.ret_sound],
                    rf=self.ret_force.clone(), rs=self.ret_stop.clone(), pops=[p.state() for p in self.pops()],
                    block=self.motor.block.clone(),
                    touch=[x.clone() for x in self.touch.state] if self.touch is not None else None,
                    rt=[x.clone() for x in self.ret_touch])

    def restore(self, sn):
        self.sound.state = [x.clone() for x in sn['sound']]
        self.ret_sound = [x.clone() for x in sn['ret']]
        self.ret_force, self.ret_stop = sn['rf'].clone(), sn['rs'].clone()
        for p, st in zip(self.pops(), sn['pops']):
            p.load(st)
        self.motor.block = sn['block'].clone()
        if self.touch is not None:
            self.touch.state = [x.clone() for x in sn['touch']]
            self.ret_touch = [x.clone() for x in sn['rt']]

    def snapshot_full(self):
        sn = self.snapshot()
        sn['tables'] = [(t.w.clone(), t.out.clone(), t.inn.clone()) for t in self.plastic()]
        sn['t'] = self.t
        return sn

    def restore_full(self, sn):
        self.restore(sn)
        for t, (w, o, i) in zip(self.plastic(), sn['tables']):
            t.w, t.out, t.inn = w.clone(), o.clone(), i.clone()
        self.t = sn['t']
        self.rlog.clear()                                   # changes made after the snapshot are gone

    def counts(self):
        """per-tick activity counts (GPU tensor; fetched in bulk by the harness)"""
        v = [x.sum() for x in self.sound.state]
        v += [self.pfc.L1ext.on.sum(), self.pfc.x_int.sum(), self.pfc.L1.on.sum(), self.pfc.L2.on.sum(),
              self.pfc.L3.on.sum(), self.pfc.L4.on.sum()]
        if self.hip is not None:
            v += [self.hip.H1.on.sum(), self.hip.h_int.sum(), self.hip.Hs.on.sum()]
        v += [self.motor.force.on.sum(), self.motor.stop.on.sum()]
        if self.touch is not None:
            v += [x.sum() for x in self.touch.state]
        return torch.stack(v)

    def count_names(self):
        n = [f'A{l + 1}' for l in range(len(self.sound.n))] + ['L1ext', 'L1int', 'L1', 'L2', 'L3', 'L4']
        if self.hip is not None:
            n += ['H1', 'H1int', 'Hs']
        n += ['force', 'stop']
        if self.touch is not None:
            n += [f'T{l + 1}' for l in range(len(self.touch.n))]
        return n
