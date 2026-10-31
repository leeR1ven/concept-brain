# -*- coding: utf-8 -*-
"""ProbeWorld = the bench World + SILENT-SCENE probes (test of self-initiated speech).  Documented addition, not a
change of the bench: the original dialogues, their content stream (rng), timing and metrics are untouched; a probe is
inserted before the first line of a dialogue with probability ss_p, decided by a SEPARATE rng.

A probe:  the dialogue's scene appears (after the usual 5 silent ticks without a scene); the partner says nothing and
waits up to ss_wait ticks.  mode 'plain': no pointing;  mode 'point': the partner points silently at one object.
  * the agent speaks (a bout = until 5 silent ticks after its stop, as a free reply):  the partner reacts at once
    (contingent reaction, like a recast):  reply == the name of an object in the scene ('point': the pointed one), or
    是X / 这是X  ->  reward 1 on the first tick and the partner repeats X;  otherwise reward 0 and the partner says the
    name of the pointed object ('point') / of a random object in the scene ('plain').
  * the agent stays silent for ss_wait ticks: no reaction.
  then the scene stays, 5 silent ticks (the usual lead), and the dialogue runs as drawn.
Record kind 'self_init' (not a free reply: evaluate() ignores it; ss_summary() scores it): reply, latency from the
scene onset, reward, overlap with the partner, acceptable set."""
from __future__ import annotations

import os
import sys

import numpy as np

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_copy")
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
from world import World, Track          # noqa: E402


class ProbeWorld(World):
    def __init__(self, stage='B', seed=0, ss_p=0.0, ss_mode='plain', ss_wait=30, **kw):
        super().__init__(stage, seed=seed, **kw)
        self.ss_p, self.ss_mode, self.ss_wait = float(ss_p), ss_mode, int(ss_wait)
        self.ss_rng = np.random.default_rng(seed * 1009 + 77 + self.dlg_offset)

    def _silent_scene(self, scene):
        p = self.p
        objs = [(s, it[0]) for s, it in enumerate(scene) if it is not None]
        if not objs:
            return
        att = None
        if self.ss_mode == 'point':
            att, X = objs[int(self.ss_rng.integers(len(objs)))]
            names = [X]
        else:
            names = [o for _, o in objs]
        acc = sorted({v for X in names for v in (X, '是' + X, '这是' + X)})
        rec = self._record(dict(stage=self._cur_stage, task='name_spont', prompt='', attend=att, item=None), 'self_init')
        self._set_scene(scene, att)
        rec['scene'] = scene; rec['acceptable'] = acc; rec['t_prompt_end'] = self.t
        tr_all = Track(); self.trk.append(tr_all)
        tr = Track(); self.trk.append(tr)
        t0 = self.t
        while True:
            self._tick()
            el = self.t - t0
            if tr.first is not None and tr.silent >= p['pause_after'] + 1:
                break
            if tr.first is None and el >= self.ss_wait:
                break
            if el >= self.ss_wait + 30:
                break
        self._finish(rec, tr)
        self.trk.remove(tr)
        rec['reward'] = 0.0
        if rec['reply']:
            good = rec['reply'] in acc
            if good:
                X = next(n for n in names if n in rec['reply'])
            else:
                X = names[int(self.ss_rng.integers(len(names)))]
            rew = p['reward_ok'] if good else p['reward_wrong']
            rec['reward'], rec['recast'] = rew, X
            self._say('other', X, reward_first=rew)
            self._silence(p['pause_after'])
        self.trk.remove(tr_all)
        rec['mouth_while_partner'] = tr_all.overlap
        rec['mouth_during_prompt'] = 0
        rec['t_end'] = self.t
        self._set_scene(scene, None)
        self.log.append(rec)

    def _run_dialogue(self, dlg):
        if self.ss_p > 0 and self.ss_rng.random() < self.ss_p and dlg['events'] and dlg['events'][0][0] == 'scene' \
                and dlg['events'][0][1] is not None:
            p = self.p
            self._cur_stage = dlg['stage']
            scene = dlg['events'][0][1]
            self._set_scene(None)
            self._silence(max(0, p['gap_dialogue'] - p['scene_lead']))
            self._silent_scene(scene)
            self._silence(p['scene_lead'])
            rest = dict(dlg); rest['events'] = [('scene_keep', scene)] + list(dlg['events'][1:])
            return self._run_rest(rest)
        return super()._run_dialogue(dlg)

    def _run_rest(self, dlg):
        """the drawn dialogue after the probe (its first scene is already shown)."""
        evs = [e for e in dlg['events'] if e[0] != 'scene_keep']
        self._cur_stage = dlg['stage']
        p = self.p
        for ev in evs:
            typ = ev[0]
            if typ == 'scene':
                self._set_scene(None)
                self._silence(max(0, p['gap_dialogue'] - p['scene_lead']))
                self._set_scene(ev[1])
            elif typ == 'change':
                self._silence(p['gap_change'] // 2)
                self._set_scene(ev[1])
                self._refs['change'] = self.t
                self._silence(p['gap_change'] - p['gap_change'] // 2)
            elif typ == 'ex':
                self._exchange(ev[1])
            elif typ == 'stmt':
                self._statement(ev[1])
            elif typ == 'play':
                self._play(ev[1])
        self.n_dialogues += 1


def ss_summary(recs, nb=4):
    """self-initiated naming: rate of speaking, accuracy (named an object of the scene), latency, reward, overlap;
    per quarter of the probes (learning)."""
    def one(rs):
        if not rs:
            return {}
        spoke = [r for r in rs if r['reply']]
        lat = [r['latency'] for r in spoke if r.get('latency') is not None]
        return dict(n=len(rs), spoke=round(len(spoke) / len(rs), 3),
                    acc=round(float(np.mean([r['reply'] in r['acceptable'] for r in rs])), 3),
                    acc_if_spoke=round(float(np.mean([r['reply'] in r['acceptable'] for r in spoke])), 3) if spoke else None,
                    lat=round(float(np.mean(lat)), 1) if lat else None,
                    overlap=int(sum(r.get('mouth_while_partner', 0) for r in rs)),
                    examples=[r['reply'] for r in spoke[-6:]])
    out = dict(all=one(recs))
    k = max(1, len(recs) // nb)
    out['quarters'] = [one(recs[i * k:(i + 1) * k]) for i in range(nb)]
    for q in out['quarters']:
        q.pop('examples', None)
    return out
